"""Maze environment — discrete 2D grid with walls, start, and goal.

TBT context
-----------
The maze is the column's sensory world. Each cell is a position in the
allocentric reference frame (L6 grid cells). The agent's movement commands
are motor efference copies that path-integrate the frame. The local sensor
reading (walls N/S/E/W, is_goal) is what L4 observes at each frame position.

The environment is external to the column — it is not learned. The column
learns an INTERNAL MODEL of the maze through experience.

Wall collision handling
-----------------------
When the agent tries to move into a wall, the environment keeps the agent
in place and returns hit_wall=True. The brain uses this signal to reverse
the frame displacement (undo the failed efference copy). See brain.py.
"""
from __future__ import annotations

import numpy as np

# Action indices
N, S, E, W = 0, 1, 2, 3
ACTION_NAMES = {N: 'N', S: 'S', E: 'E', W: 'W'}

# Displacement vectors: (d_row, d_col)
DELTA: dict[int, tuple[int, int]] = {
    N: (-1,  0),
    S: ( 1,  0),
    E: ( 0,  1),
    W: ( 0, -1),
}

# Default 5×5 maze (hard-coded for reproducibility).
# 0 = open cell, 1 = wall.
# Start = (0, 0) top-left, Goal = (4, 4) bottom-right.
# Deliberately includes dead-ends and interior walls so that
# cells with identical sensor readings exist — this forces the
# column to rely on path integration for disambiguation.
#
# Layout (row 0 = top):
#   S . # . .
#   . # . # .
#   . . . . #
#   # . # . .
#   . . . # G
#
DEFAULT_MAZE = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
], dtype=np.uint8)

DEFAULT_START = (0, 0)
DEFAULT_GOAL  = (4, 4)


class MazeEnv:
    """Discrete 2-D grid maze.

    Parameters
    ----------
    grid  : (H, W) uint8 array — 0=open, 1=wall
    start : (row, col) of starting cell
    goal  : (row, col) of goal cell
    """

    def __init__(
        self,
        grid:  np.ndarray = DEFAULT_MAZE,
        start: tuple[int, int] = DEFAULT_START,
        goal:  tuple[int, int] = DEFAULT_GOAL,
    ) -> None:
        self.grid  = grid.copy()
        self.start = start
        self.goal  = goal
        self.H, self.W = grid.shape

        assert grid[start] == 0, f"Start {start} is a wall"
        assert grid[goal]  == 0, f"Goal {goal} is a wall"

        self.pos:      tuple[int, int] = start
        self.prev_pos: tuple[int, int] = start

    # ------------------------------------------------------------------
    # Episode control
    # ------------------------------------------------------------------

    def reset(self) -> tuple[int, int]:
        """Reset to default start position."""
        self.pos      = self.start
        self.prev_pos = self.start
        return self.pos

    def reset_at(self, pos: tuple[int, int]) -> tuple[int, int]:
        """Reset to an arbitrary open cell."""
        assert self.grid[pos] == 0, f"Position {pos} is a wall"
        self.pos      = pos
        self.prev_pos = pos
        return self.pos

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: int) -> tuple[tuple[int, int], bool]:
        """Take one step.

        Returns
        -------
        new_pos  : (row, col) after the step
        hit_wall : True if the move was blocked by a wall

        The agent stays in place when it hits a wall; hit_wall=True lets
        the brain reverse the corresponding frame displacement (undo the
        failed efference copy).
        """
        dr, dc = DELTA[action]
        r, c   = self.pos
        nr, nc = r + dr, c + dc

        self.prev_pos = self.pos

        if 0 <= nr < self.H and 0 <= nc < self.W and self.grid[nr, nc] == 0:
            self.pos = (nr, nc)
            return self.pos, False
        else:
            return self.pos, True   # wall or out-of-bounds

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def valid_actions(self) -> list[int]:
        """Actions that do not hit a wall from the current position."""
        r, c = self.pos
        result = []
        for a, (dr, dc) in DELTA.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.H and 0 <= nc < self.W and self.grid[nr, nc] == 0:
                result.append(a)
        return result

    def reached_goal(self) -> bool:
        return self.pos == self.goal

    def open_cells(self) -> list[tuple[int, int]]:
        """All open (non-wall) cell positions."""
        return [
            (r, c)
            for r in range(self.H)
            for c in range(self.W)
            if self.grid[r, c] == 0
        ]

    def n_open(self) -> int:
        return int((self.grid == 0).sum())

    def reachable_cells(self, start: tuple[int, int] | None = None) -> list[tuple[int, int]]:
        """BFS from start (default: self.start) — returns all reachable open cells."""
        origin = start if start is not None else self.start
        visited = {origin}
        queue = [origin]
        while queue:
            r, c = queue.pop(0)
            for dr, dc in DELTA.values():
                nr, nc = r + dr, c + dc
                if ((nr, nc) not in visited
                        and 0 <= nr < self.H and 0 <= nc < self.W
                        and self.grid[nr, nc] == 0):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return list(visited)

    def n_reachable(self, start: tuple[int, int] | None = None) -> int:
        return len(self.reachable_cells(start))

    def nearest_unvisited(self, src: tuple[int, int],
                          visited: set[tuple[int, int]]) -> list[int] | None:
        """BFS from src to the nearest open cell not in visited.

        Returns the action sequence to reach it, or None if all reachable
        cells have been visited.
        """
        if not visited:
            return []
        parent: dict[tuple, tuple | None] = {src: None}
        action_to: dict[tuple, int] = {}
        queue = [src]
        while queue:
            pos = queue.pop(0)
            if pos not in visited:
                # Reconstruct path back to src
                path = []
                cur = pos
                while cur != src:
                    path.append(action_to[cur])
                    cur = parent[cur]
                path.reverse()
                return path
            r, c = pos
            for a, (dr, dc) in DELTA.items():
                npos = (r + dr, c + dc)
                if (npos not in parent
                        and 0 <= npos[0] < self.H and 0 <= npos[1] < self.W
                        and self.grid[npos] == 0):
                    parent[npos] = pos
                    action_to[npos] = a
                    queue.append(npos)
        return None   # all reachable cells visited

    def bfs_path(self, src: tuple[int, int],
                 dst: tuple[int, int]) -> list[int] | None:
        """Return list of actions for shortest path src→dst, or None if unreachable."""
        if src == dst:
            return []
        parent: dict[tuple, tuple | None] = {src: None}
        action_to: dict[tuple, int] = {}
        queue = [src]
        while queue:
            pos = queue.pop(0)
            r, c = pos
            for a, (dr, dc) in DELTA.items():
                npos = (r + dr, c + dc)
                if (npos not in parent
                        and 0 <= npos[0] < self.H and 0 <= npos[1] < self.W
                        and self.grid[npos] == 0):
                    parent[npos] = pos
                    action_to[npos] = a
                    if npos == dst:
                        # Reconstruct path
                        path = []
                        cur = dst
                        while cur != src:
                            path.append(action_to[cur])
                            cur = parent[cur]
                        path.reverse()
                        return path
                    queue.append(npos)
        return None   # unreachable

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, overlay: dict[tuple, str] | None = None) -> str:
        """ASCII render of the maze.

        overlay: optional dict mapping (row, col) → single char to draw.
        Agent position is shown as 'A', goal as 'G', start as 'S',
        walls as '#', open cells as '.'.
        """
        overlay = overlay or {}
        lines = []
        for r in range(self.H):
            row = []
            for c in range(self.W):
                pos = (r, c)
                if pos in overlay:
                    row.append(overlay[pos])
                elif self.grid[r, c] == 1:
                    row.append('#')
                elif pos == self.pos:
                    row.append('A')
                elif pos == self.goal:
                    row.append('G')
                elif pos == self.start:
                    row.append('S')
                else:
                    row.append('.')
            lines.append(' '.join(row))
        return '\n'.join(lines)
