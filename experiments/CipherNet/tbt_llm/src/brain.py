"""SingleColumnBrain — TBT single-column spatial agent.

Architecture
------------
  AllocentricFrame (L6) — exact path integration on a discrete grid
  MiniColumn       (L4) — position → SDR union model (learn_one / _model lookup)

Action selection priority
-------------------------
  1. Goal adjacent → move there (pragmatic value)
  2. Unmapped adjacent → explore (epistemic value; exact _model lookup)
  3. Otherwise → random walk

When all adjacent cells are already mapped and the goal is not adjacent,
the column has no local information signal and random-walks.  This is
correct single-column behaviour.  Systematic long-range exploration
(hippocampal replay, place-cell sequences) lies outside a single column's
competence and is not implemented here.

Phase 2 (localisation from unknown start) requires multi-column voting
and is marked N/A in the scaling experiment.

Note on MiniColumn.predict()
-----------------------------
MiniColumn.predict() uses ±1 neighbour search designed for continuous
visual space (sub-pixel centroid jitter).  For the discrete maze grid we
use `mini_column._model.get(next_pos)` for exact position lookup so that
only genuinely mapped cells return a hit.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_CIPHER_SRC = _SRC.parent.parent / 'src'
if str(_CIPHER_SRC) not in sys.path:
    sys.path.insert(0, str(_CIPHER_SRC))

from reference_frames import AllocentricFrame
from column import MiniColumn
from maze_env import MazeEnv, DELTA
from sensor import LocalSensor

# Module-level sensor — experiment scripts may swap this for DistanceSensor
_SENSOR: LocalSensor = LocalSensor()


class SingleColumnBrain:
    """One TBT cortical column navigating a maze.

    Parameters
    ----------
    goal      : maze cell the agent is trying to reach (or None)
    epsilon   : random-action probability (0 = pure policy)

    The following legacy parameters are accepted and silently ignored so
    that old experiment scripts that pass them do not break:
        confidence_threshold, curiosity_weight, goal_weight,
        n_cells, min_coverage
    """

    def __init__(
        self,
        goal: Optional[tuple[int, int]] = None,
        epsilon: float = 0.0,
        # Legacy / ignored parameters kept for API compatibility
        confidence_threshold: float = 1.0,
        curiosity_weight: float = 1.0,
        goal_weight: float = 2.0,
        n_cells: int = 0,
        min_coverage: float = 0.0,
    ) -> None:
        self.mini_column = MiniColumn()
        self.frame       = AllocentricFrame(position=(0.0, 0.0), resolution=1.0)
        self.goal        = goal
        self.epsilon     = epsilon

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def reset(self, start_pos: tuple[int, int],
              known_start: bool = True) -> None:
        """Begin a new episode.

        The MiniColumn model is NOT reset — it persists across episodes.
        known_start is accepted for API compatibility but has no effect:
        single-column localisation from unknown start is N/A (requires
        multi-column voting).
        """
        self.frame.set_position((float(start_pos[0]), float(start_pos[1])))

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: int, env: MazeEnv) -> np.ndarray:
        """Execute action; update path integration and map.

        1. Motor command → env
        2. Efference copy → AllocentricFrame update (undone if wall hit)
        3. Observe sensor → write (sdr, pos) to MiniColumn

        Returns the observed SDR.
        """
        _, hit_wall = env.step(action)

        dr, dc = DELTA[action]
        self.frame.update((float(dr), float(dc)))
        if hit_wall:
            self.frame.update((-float(dr), -float(dc)))

        sdr = _SENSOR.encode(env)
        self.mini_column.learn_one(sdr, self.frame.position_key())
        return sdr

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, valid_actions: list[int]) -> int:
        """Choose action by TBT single-column priority.

        Priority order:
          1. Goal adjacent   — move to goal if reachable in one step
          2. Unmapped cell   — explore (exact _model lookup, no neighbour blur)
          3. Random walk     — honest fallback when nothing is locally novel
        """
        if not valid_actions:
            raise ValueError("No valid actions")

        if self.epsilon > 0 and random.random() < self.epsilon:
            return random.choice(valid_actions)

        pos = self.frame.position_key()

        # Priority 1: step onto goal
        if self.goal is not None:
            for a in valid_actions:
                dr, dc = DELTA[a]
                if (pos[0] + dr, pos[1] + dc) == self.goal:
                    return a

        # Priority 2: step into an unmapped cell (exact lookup)
        unmapped = [
            a for a in valid_actions
            if self.mini_column._model.get(
                (pos[0] + DELTA[a][0], pos[1] + DELTA[a][1])
            ) is None
        ]
        if unmapped:
            return random.choice(unmapped)

        # Priority 3: random walk — single column has no long-range plan
        return random.choice(valid_actions)

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------

    def n_mapped(self) -> int:
        """Number of distinct positions written to the model."""
        return self.mini_column.n_locations()

    def coverage(self, n_cells: int) -> float:
        """Fraction of maze cells mapped (caller supplies total cell count)."""
        return self.mini_column.n_locations() / max(1, n_cells)

    # ------------------------------------------------------------------
    # Localisation stubs (N/A for single column)
    # ------------------------------------------------------------------

    def observe(self, sdr: np.ndarray) -> None:
        """Write sdr at current frame position (no-op belief update)."""
        self.mini_column.learn_one(sdr, self.frame.position_key())

    def best_estimate(self) -> Optional[tuple]:
        """Single-column localisation is N/A; returns frame position."""
        return self.frame.position_key()

    def belief_entropy(self) -> float:
        """Single-column belief entropy is N/A; always returns inf."""
        return float('inf')

    def is_localised(self) -> bool:
        return False
