"""Procedural LockPath layouts — the data distribution for the learning agent.

The four hand-authored LockPath levels are demo content; the binding experiment
needs *distributions* of layouts per mechanic, with a train/held-out split, so we
can measure whether a model generalizes to unseen positions of the same mechanic
(EXPERIMENT_GOALS P2 — the shift-invisibility test).

Each generator samples a layout (random room size and element positions) for one
mechanic. Layouts are emitted as `List[str]` in the LockPath ASCII format and are
**validated by the BFS oracle** — only solvable layouts are returned. The
generators stay distributional and template-driven; no per-instance hand-tuning.

Mechanics:
  nav        reach the goal (no obstacles)
  key_door   a wall column with a door gap; key on the agent's side
  block_pad  push a block onto a pad, then reach the goal
  compose    key+door AND block+pad together (the in-distribution A∘B mechanic)
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

from arc_agi_3.games import LockPath
from arc_agi_3.oracle import is_solvable

Layout = List[str]
Cell = Tuple[int, int]


# --- low-level board helpers ------------------------------------------------


def _room(w: int, h: int) -> List[List[str]]:
    """A rectangular room: border walls, floor interior."""
    grid = [["." for _ in range(w)] for _ in range(h)]
    for x in range(w):
        grid[0][x] = grid[h - 1][x] = "#"
    for y in range(h):
        grid[y][0] = grid[y][w - 1] = "#"
    return grid


def _interior(w: int, h: int) -> List[Cell]:
    return [(x, y) for x in range(1, w - 1) for y in range(1, h - 1)]


def _to_rows(grid: List[List[str]]) -> Layout:
    return ["".join(row) for row in grid]


def _place(grid: List[List[str]], cell: Cell, ch: str) -> None:
    x, y = cell
    grid[y][x] = ch


def _pop(rng: random.Random, pool: List[Cell]) -> Cell:
    """Remove and return a random cell from `pool`."""
    i = rng.randrange(len(pool))
    return pool.pop(i)


# --- per-mechanic generators ------------------------------------------------


def gen_nav(rng: random.Random) -> Optional[Layout]:
    w, h = rng.randint(8, 12), rng.randint(6, 9)
    grid = _room(w, h)
    free = _interior(w, h)
    _place(grid, _pop(rng, free), "A")
    _place(grid, _pop(rng, free), "G")
    return _to_rows(grid)


def gen_key_door(rng: random.Random) -> Optional[Layout]:
    w, h = rng.randint(10, 12), rng.randint(6, 9)
    col = rng.randint(3, w - 4)            # interior wall column
    door_y = rng.randint(1, h - 2)
    grid = _room(w, h)
    for y in range(1, h - 1):              # seal the column, leaving one door gap
        _place(grid, (col, y), "D" if y == door_y else "#")
    left = [(x, y) for x in range(1, col) for y in range(1, h - 1)]
    right = [(x, y) for x in range(col + 1, w - 1) for y in range(1, h - 1)]
    if len(left) < 2 or len(right) < 1:
        return None
    _place(grid, _pop(rng, left), "A")
    _place(grid, _pop(rng, left), "K")
    _place(grid, _pop(rng, right), "G")
    return _to_rows(grid)


def gen_block_pad(rng: random.Random) -> Optional[Layout]:
    w, h = rng.randint(9, 12), rng.randint(7, 10)
    grid = _room(w, h)
    # Keep the block off the border ring so it can be pushed in any direction.
    inner = [(x, y) for x in range(2, w - 2) for y in range(2, h - 2)]
    free = _interior(w, h)
    if not inner:
        return None
    block = _pop(rng, inner)
    free.remove(block)
    _place(grid, block, "B")
    for ch in ("P", "G", "A"):
        _place(grid, _pop(rng, free), ch)
    return _to_rows(grid)


def gen_compose(rng: random.Random) -> Optional[Layout]:
    w, h = rng.randint(11, 12), rng.randint(7, 9)
    col = rng.randint(3, 4)                # narrow left room (agent + key)
    door_y = rng.randint(1, h - 2)
    grid = _room(w, h)
    for y in range(1, h - 1):
        _place(grid, (col, y), "D" if y == door_y else "#")
    left = [(x, y) for x in range(1, col) for y in range(1, h - 1)]
    right = [(x, y) for x in range(col + 2, w - 2) for y in range(2, h - 2)]
    right_free = [(x, y) for x in range(col + 1, w - 1) for y in range(1, h - 1)]
    if len(left) < 2 or len(right) < 1 or len(right_free) < 2:
        return None
    _place(grid, _pop(rng, left), "A")
    _place(grid, _pop(rng, left), "K")
    block = _pop(rng, right)               # block off the right-room border
    right_free.remove(block)
    _place(grid, block, "B")
    for ch in ("P", "G"):
        _place(grid, _pop(rng, right_free), ch)
    return _to_rows(grid)


GENERATORS: Dict[str, Callable[[random.Random], Optional[Layout]]] = {
    "nav": gen_nav,
    "key_door": gen_key_door,
    "block_pad": gen_block_pad,
    "compose": gen_compose,
}


# --- sampling & splitting ---------------------------------------------------


def sample_layouts(
    mechanic: str,
    n: int,
    seed: int = 0,
    max_attempts: int = 10000,
) -> List[Layout]:
    """Sample `n` distinct, BFS-solvable layouts for `mechanic` (deterministic)."""
    if mechanic not in GENERATORS:
        raise ValueError(f"unknown mechanic {mechanic!r}; choose from {list(GENERATORS)}")
    gen = GENERATORS[mechanic]
    rng = random.Random(seed)
    out: List[Layout] = []
    seen = set()
    attempts = 0
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        layout = gen(rng)
        if layout is None:
            continue
        key = tuple(layout)
        if key in seen:
            continue
        if not is_solvable(_loaded_game(layout)):
            continue
        seen.add(key)
        out.append(layout)
    if len(out) < n:
        raise RuntimeError(
            f"only generated {len(out)}/{n} {mechanic} layouts in {max_attempts} attempts"
        )
    return out


def train_test_split(
    layouts: List[Layout], train_frac: float = 0.7
) -> Tuple[List[Layout], List[Layout]]:
    """Split a sampled pool into disjoint train / held-out sets by index.

    The held-out layouts are unseen position-configs of the same mechanic — the
    distribution-shift the binding experiment measures generalization against.
    """
    cut = int(round(len(layouts) * train_frac))
    return layouts[:cut], layouts[cut:]


def make_game(layouts: List[Layout]) -> LockPath:
    """A LockPath whose levels are the given procedural layouts."""
    return LockPath(levels=list(layouts))


def _loaded_game(layout: Layout) -> LockPath:
    """A single-level LockPath with level 0 loaded — ready for the oracle.

    The oracle reads live game state, so the level must be loaded first;
    `LockPath([layout])` alone leaves the default (empty) state.
    """
    game = LockPath([layout])
    game.load_level(0)
    return game
