"""Local sensor — SDR encoding the agent's immediate surroundings.

Two sensor types are provided:

LocalSensor (5-bit binary, original)
-------------------------------------
Minimal sensor used in Experiment 1. Gives the column just enough signal to
distinguish cell TYPES (corner, corridor, junction, goal) but not cell
POSITIONS. Identical readings at many interior cells forces the column to rely
entirely on path integration for disambiguation. This is the core TBT test.

  bit 0 — wall to North  (row-1 is wall or boundary)
  bit 1 — wall to South  (row+1 is wall or boundary)
  bit 2 — wall to East   (col+1 is wall or boundary)
  bit 3 — wall to West   (col-1 is wall or boundary)
  bit 4 — is_goal        (current cell == goal)

Shape: (5,) int8.  All bits 0 or 1.

DistanceSensor (9-value extended, for larger mazes)
-----------------------------------------------------
Motivated by the neuroscience finding that the brain uses LOG-COMPRESSED
distance representations (Weber-Fechner law, log-domain encoding). Instead
of binary wall-or-not, this sensor encodes how far away the nearest wall is
in each direction, log-compressed to a small integer.

Encoding: for each direction d in {N, S, E, W}:
  look ahead until hitting a wall or boundary
  dist = number of open cells before the wall (0 = wall immediately adjacent)
  encode as floor(log2(dist + 1)) clipped to [0, max_val]

With max_val=3 this gives 2 bits per direction:
  0 = wall is the next cell (immediate wall)
  1 = 1 open cell then wall (short corridor)
  2 = 2-3 open cells then wall (medium corridor)
  3 = 4+ open cells (long corridor or open space)

Plus 1 binary is_goal bit = 9 values total.
Shape: (9,) int8.  Values in [0, max_val] or {0,1} for goal bit.

The log compression is motivated by Nieder & Miller (2003) and the LGMD
neuron result (Gabbiani et al. 2002): the brain encodes magnitudes
logarithmically, and this is computationally sufficient for ratio operations
(which dominate spatial navigation). In a large maze with long corridors,
binary wall detection gives ambiguous readings; log-compressed distance
provides much more discrimination power without requiring the column to
know its absolute position.
"""
from __future__ import annotations

import math

import numpy as np

from maze_env import MazeEnv, N, S, E, W, DELTA

# ---- Binary sensor constants ----
BIT_N    = 0
BIT_S    = 1
BIT_E    = 2
BIT_W    = 3
BIT_GOAL = 4
N_BITS   = 5

# ---- Distance sensor constants ----
DIST_N     = 0
DIST_S     = 1
DIST_E     = 2
DIST_W     = 3
DIST_GOAL  = 8
N_DIST_BITS = 9
DIST_MAX_VAL = 3   # floor(log2(dist+1)) clipped here


class LocalSensor:
    """Encodes the agent's local observations as a 5-bit binary SDR."""

    def encode(self, env: MazeEnv) -> np.ndarray:
        """Return a (5,) int8 array for the current agent position."""
        r, c         = env.pos
        n_rows, n_cols = env.H, env.W
        sdr            = np.zeros(N_BITS, dtype=np.int8)

        # Wall bits: 1 if the neighbour in that direction is a wall or boundary
        # Note: local H/W names avoided because W=3 is the action constant (West).
        for bit, (dr, dc) in [(BIT_N, DELTA[N]), (BIT_S, DELTA[S]),
                              (BIT_E, DELTA[E]), (BIT_W, DELTA[W])]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n_rows and 0 <= nc < n_cols) or env.grid[nr, nc] == 1:
                sdr[bit] = 1

        sdr[BIT_GOAL] = 1 if env.pos == env.goal else 0
        return sdr

    def encode_at(self, env: MazeEnv, pos: tuple[int, int]) -> np.ndarray:
        """Encode as if the agent were at `pos` (does not move the agent)."""
        saved = env.pos
        env.pos = pos
        sdr = self.encode(env)
        env.pos = saved
        return sdr


class DistanceSensor:
    """Extended sensor encoding log-compressed wall distances.

    Motivated by the neuroscience of log-domain spatial representations.
    Provides much better discrimination in large mazes where the binary
    sensor produces many ambiguous readings.

    Shape: (9,) int8.  Values: [dist_N, dist_S, dist_E, dist_W,
                                 pad, pad, pad, pad, is_goal]
    where each dist is floor(log2(lookahead_distance + 1)) in [0, DIST_MAX_VAL].
    """

    def _lookahead(self, env: MazeEnv, r: int, c: int,
                   dr: int, dc: int) -> int:
        """Count open cells in direction (dr,dc) before hitting wall/boundary."""
        n_rows, n_cols = env.H, env.W
        dist = 0
        nr, nc = r + dr, c + dc
        while (0 <= nr < n_rows and 0 <= nc < n_cols
               and env.grid[nr, nc] == 0):
            dist += 1
            nr += dr
            nc += dc
        return dist

    def _log_compress(self, dist: int) -> int:
        return min(int(math.floor(math.log2(dist + 1))), DIST_MAX_VAL)

    def encode(self, env: MazeEnv) -> np.ndarray:
        """Return a (9,) int8 array for the current agent position."""
        r, c = env.pos
        sdr  = np.zeros(N_DIST_BITS, dtype=np.int8)
        for i, (action, (dr, dc)) in enumerate(
                [(N, DELTA[N]), (S, DELTA[S]), (E, DELTA[E]), (W, DELTA[W])]):
            dist = self._lookahead(env, r, c, dr, dc)
            sdr[i] = self._log_compress(dist)
        sdr[DIST_GOAL] = 1 if env.pos == env.goal else 0
        return sdr

    def encode_at(self, env: MazeEnv, pos: tuple[int, int]) -> np.ndarray:
        """Encode as if the agent were at `pos` (does not move the agent)."""
        saved = env.pos
        env.pos = pos
        sdr = self.encode(env)
        env.pos = saved
        return sdr
