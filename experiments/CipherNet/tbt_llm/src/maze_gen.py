"""Procedural maze generation for TBT experiments.

Generates mazes of arbitrary size using recursive-backtracking DFS, which
produces perfect mazes (exactly one path between any two cells, no isolated
regions). All cells are reachable from start by construction.

Difficulty controls
-------------------
  size      : NxN grid (larger = harder by default)
  braid_p   : probability of removing a dead-end wall to create loops
              0.0 = perfect maze (single path everywhere, maximum backtracking)
              1.0 = fully braided (many paths, minimal dead-ends)
  extra_walls: after generation, randomly seal fraction of open passages back
              into walls. Creates disconnected pockets and longer detours.

Ambiguity metric
----------------
After generation, compute the fraction of open cells that share their 5-bit
binary sensor SDR with at least one other cell. High ambiguity means the path
integrator is doing more of the work — this is the quantity we care about for
TBT experiments.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np

from maze_env import MazeEnv, N, S, E, W, DELTA


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def _opposite(action: int) -> int:
    return {N: S, S: N, E: W, W: E}[action]


def generate_maze(
    rows: int,
    cols: int,
    start: tuple[int, int] = (0, 0),
    goal: Optional[tuple[int, int]] = None,
    braid_p: float = 0.0,
    seed: Optional[int] = None,
) -> MazeEnv:
    """Generate a perfect maze via recursive-backtracking DFS.

    Parameters
    ----------
    rows, cols  : grid dimensions (cells, not corridors)
    start       : starting cell (row, col)
    goal        : goal cell; defaults to (rows-1, cols-1)
    braid_p     : probability of removing each dead-end wall (0=perfect, 1=open)
    seed        : random seed for reproducibility

    Returns
    -------
    MazeEnv with grid, start, and goal set.
    """
    rng = random.Random(seed)
    if goal is None:
        goal = (rows - 1, cols - 1)

    # Start with all walls
    grid = np.ones((rows, cols), dtype=np.uint8)

    # Iterative recursive-backtracking: carve passages.
    # Iterative (not recursive) to avoid Python's default 1000-frame stack
    # limit, which is exceeded at ~32x32 and above.
    # Each stack entry stores the cell and its remaining shuffled directions
    # so that the random branching order is preserved exactly.
    visited: set[tuple[int, int]] = set()
    visited.add((start[0], start[1]))
    grid[start[0], start[1]] = 0
    dirs0 = [N, S, E, W]
    rng.shuffle(dirs0)
    stack: list[tuple[int, int, list[int]]] = [(start[0], start[1], dirs0)]

    while stack:
        r, c, dirs = stack[-1]
        if not dirs:
            stack.pop()
            continue
        action = dirs.pop()
        dr, dc = DELTA[action]
        nr, nc = r + dr, c + dc
        if (0 <= nr < rows and 0 <= nc < cols
                and (nr, nc) not in visited):
            visited.add((nr, nc))
            grid[nr, nc] = 0
            new_dirs = [N, S, E, W]
            rng.shuffle(new_dirs)
            stack.append((nr, nc, new_dirs))

    # Braiding: remove dead-end walls to create loops
    if braid_p > 0.0:
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 0:
                    # Check if this is a dead-end (only one open neighbour)
                    open_neighbours = []
                    wall_neighbours = []
                    for action, (dr, dc) in DELTA.items():
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if grid[nr, nc] == 0:
                                open_neighbours.append((nr, nc))
                            else:
                                wall_neighbours.append((nr, nc))
                    if len(open_neighbours) == 1 and wall_neighbours:
                        if rng.random() < braid_p:
                            nr, nc = rng.choice(wall_neighbours)
                            grid[nr, nc] = 0

    # Ensure start and goal are open
    grid[start] = 0
    grid[goal]  = 0

    return MazeEnv(grid=grid, start=start, goal=goal)


# ---------------------------------------------------------------------------
# Sensor ambiguity analysis
# ---------------------------------------------------------------------------

def sensor_ambiguity(env: MazeEnv) -> float:
    """Fraction of open cells that share their 5-bit SDR with another cell.

    Higher ambiguity = more reliance on path integration for disambiguation.
    0.0 = every cell has a unique sensor reading (sensor alone suffices).
    1.0 = every cell's reading is shared with at least one other cell.
    """
    from sensor import LocalSensor
    sensor = LocalSensor()
    readings: dict[tuple, list] = {}
    for pos in env.open_cells():
        key = tuple(sensor.encode_at(env, pos).tolist())
        readings.setdefault(key, []).append(pos)
    ambiguous = sum(len(v) for v in readings.values() if len(v) > 1)
    total = len(env.open_cells())
    return ambiguous / total if total > 0 else 0.0


def maze_stats(env: MazeEnv) -> dict:
    """Summary statistics for a maze."""
    reachable = env.reachable_cells()
    path = env.bfs_path(env.start, env.goal)
    return {
        'size'           : (env.H, env.W),
        'open_cells'     : env.n_open(),
        'reachable'      : len(reachable),
        'shortest_path'  : len(path) if path is not None else None,
        'ambiguity'      : sensor_ambiguity(env),
    }


# ---------------------------------------------------------------------------
# Named difficulty presets
# ---------------------------------------------------------------------------

def make_maze(difficulty: str = 'small',
              seed: Optional[int] = None) -> MazeEnv:
    """Return a maze at the named difficulty level.

    Levels
    ------
    tiny      :  5x5  perfect maze (baseline, same as DEFAULT_MAZE shape)
    small     : 10x10 perfect maze
    medium    : 20x20 perfect maze
    large     : 40x40 perfect maze
    huge      : 80x80 perfect maze
    braided_s : 15x15 with braid_p=0.3 (many loops, less dead-ends)
    braided_l : 30x30 with braid_p=0.3
    """
    presets = {
        'tiny'     : dict(rows=5,  cols=5,  braid_p=0.0),
        'small'    : dict(rows=10, cols=10, braid_p=0.0),
        'medium'   : dict(rows=20, cols=20, braid_p=0.0),
        'large'    : dict(rows=40, cols=40, braid_p=0.0),
        'huge'     : dict(rows=80, cols=80, braid_p=0.0),
        'braided_s': dict(rows=15, cols=15, braid_p=0.3),
        'braided_l': dict(rows=30, cols=30, braid_p=0.3),
    }
    if difficulty not in presets:
        raise ValueError(f"Unknown difficulty: {difficulty!r}. "
                         f"Choose from {list(presets)}")
    kwargs = presets[difficulty]
    return generate_maze(seed=seed, **kwargs)
